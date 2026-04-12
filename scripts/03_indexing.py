import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Charge automatiquement les variables du fichier .env
load_dotenv()  # ← et ici, avant tout le reste

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INPUT_FILE      = Path(__file__).parents[1] / "data" / "raw" / "ihl_dataset.jsonl"
VECTORSTORE_DIR = Path(__file__).parents[1] / "data" / "vectorstore"
EMBEDDING_MODEL = "mistral-embed"
BATCH_SIZE      = 50  # Mistral recommande des batches plus petits qu'OpenAI

# ── Chargement des chunks ─────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[Document]:
    documents = []
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

# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    log.info("=== Démarrage de l'indexing ===\n")

    # Vérification clé API
    if not os.getenv("MISTRAL_API_KEY"):
        log.error("❌ MISTRAL_API_KEY non définie.")
        log.error("   Lance : export MISTRAL_API_KEY='...'")
        return

    # Vérification fichier d'entrée
    if not INPUT_FILE.exists():
        log.error(f"❌ Fichier introuvable : {INPUT_FILE}")
        log.error("   Lance d'abord : python scripts/02_chunking.py")
        return

    # Chargement des chunks
    log.info(f"Chargement des chunks depuis {INPUT_FILE}...")
    documents = load_chunks(INPUT_FILE)
    log.info(f"  ✓ {len(documents)} chunks chargés\n")

    # Initialisation du modèle d'embeddings
    log.info(f"Modèle d'embeddings : {EMBEDDING_MODEL}")
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    # Création du vectorstore FAISS par batches
    log.info(f"Embeddings + indexing par batches de {BATCH_SIZE}...\n")
    vectorstore = None

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        log.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    # Sauvegarde du vectorstore
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    log.info(f"\n=== Terminé ===")
    log.info(f"✅ Vectorstore sauvegardé → {VECTORSTORE_DIR}")
    log.info(f"   Fichiers : index.faiss + index.pkl")

    # Test rapide
    log.info("\n--- Test de recherche ---")
    query = "Qu'est-ce que le crime de génocide ?"
    results = vectorstore.similarity_search(query, k=3)
    log.info(f"Requête : '{query}'")
    for r in results:
        log.info(f"  → [{r.metadata['source'][:40]}] {r.metadata['article']} : {r.page_content[:80]}...")


if __name__ == "__main__":
    main()
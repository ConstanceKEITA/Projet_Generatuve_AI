import json
import re
import uuid
import logging
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INPUT_FILE  = Path(__file__).parents[1] / "data" / "raw" / "ihl_raw.json"
OUTPUT_FILE = Path(__file__).parents[1] / "data" / "raw" / "ihl_dataset.jsonl"

CHUNK_SIZE    = 800   # caractères max par chunk
CHUNK_OVERLAP = 100   # chevauchement entre chunks si article trop long

# ── Détection des articles ────────────────────────────────────────────────────

# Détecte : "Article 1", "Article 12 bis", "ARTICLE PREMIER", etc.
ARTICLE_RE = re.compile(
    r"(?:^|\n)\s*(Article(?:s)?\s+\w+(?:\s+bis)?|ARTICLE\s+\w+)\s*[:\-–]?\s*(.*?)(?=\n|$)",
    re.IGNORECASE | re.MULTILINE,
    )

# Détecte : "Partie II", "Chapitre III", "Section 2"
PARTIE_RE = re.compile(
    r"(?:^|\n)\s*(Partie\s+[IVXLCDM\d]+|Chapitre\s+[IVXLCDM\d]+|Section\s+\d+)\s*[:\-–]?\s*(.*?)(?=\n|$)",
    re.IGNORECASE | re.MULTILINE,
    )

def split_by_articles(text: str) -> list[dict]:
    matches = list(ARTICLE_RE.finditer(text))

    if not matches:
        return [{"article": None, "titre": None, "partie": None, "text": text}]

    blocks = []
    current_partie = None

    for i, m in enumerate(matches):
        article_label = m.group(1).strip()
        titre         = m.group(2).strip() or None
        start         = m.start()
        end           = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text    = text[start:end].strip()

        # Cherche une partie/chapitre dans le texte précédant cet article
        preceding = text[matches[i - 1].end() if i > 0 else 0 : start]
        partie_matches = list(PARTIE_RE.finditer(preceding))
        if partie_matches:
            current_partie = partie_matches[-1].group(0).strip()

        blocks.append({
            "article": article_label,
            "titre":   titre,
            "partie":  current_partie,
            "text":    block_text,
        })

    return blocks

# ── Chunking fin ──────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + size

        if end < len(text):
            # Préférer couper sur un saut de ligne
            cut = text.rfind("\n", start, end)
            if cut == -1:
                # Sinon sur une fin de phrase
                cut = text.rfind(". ", start, end)
            if cut != -1:
                end = cut + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]

# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_document(doc: dict) -> list[dict]:
    blocks = split_by_articles(doc["text"])
    chunks = []

    for block in blocks:
        sub_texts = chunk_text(block["text"])

        for idx, sub in enumerate(sub_texts):
            chunks.append({
                "id":          str(uuid.uuid4()),
                "source":      doc["name"],
                "year":        doc["year"],
                "type":        doc["type"],
                "url":         doc["url"],
                "lang":        doc["lang"],
                "tags":        doc["tags"],
                "partie":      block["partie"],
                "article":     block["article"],
                "titre":       block["titre"],
                "chunk_index": idx,
                "char_count":  len(sub),
                "text":        sub,
            })

    return chunks


def main():
    log.info("=== Démarrage du chunking ===\n")

    # Lecture du fichier brut
    if not INPUT_FILE.exists():
        log.error(f"❌ Fichier introuvable : {INPUT_FILE}")
        log.error("   Lance d'abord : python scripts/01_scraping.py")
        return

    with INPUT_FILE.open(encoding="utf-8") as f:
        documents = json.load(f)

    log.info(f"  ✓ {len(documents)} documents chargés depuis {INPUT_FILE}\n")

    # Chunking
    all_chunks = []
    for doc in documents:
        chunks = process_document(doc)
        all_chunks.extend(chunks)
        log.info(f"  → {doc['name'][:55]:<55} {len(chunks):>4} chunks")

    # Sauvegarde en JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info(f"\n=== Terminé ===")
    log.info(f"✅ {len(all_chunks)} chunks sauvegardés → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
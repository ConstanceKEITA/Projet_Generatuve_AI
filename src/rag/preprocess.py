from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def preprocess_documents(texts: List[str], chunk_size: int, overlap: int) -> List[str]:
    all_chunks = []
    for t in texts:
        all_chunks.extend(chunk_text(t, chunk_size, overlap))
    return all_chunks
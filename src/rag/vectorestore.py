from typing import List, Tuple
import numpy as np

class SimpleVectorStore:
    def __init__(self, embeddings: List[list], texts: List[str]):
        self.embeddings = np.array(embeddings)
        self.texts = texts

    def similarity_search(self, query_embedding: list, k: int = 3) -> List[Tuple[str, float]]:
        query = np.array(query_embedding)
        sims = self.embeddings @ query / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query) + 1e-8
        )
        idxs = sims.argsort()[::-1][:k]
        return [(self.texts[i], float(sims[i])) for i in idxs]
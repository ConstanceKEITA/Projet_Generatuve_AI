from typing import List
import numpy as np

def embed_texts(texts: List[str]) -> List[list]:
    """Mock d'embeddings (à remplacer par un vrai modèle)."""
    return [np.random.rand(384).tolist() for _ in texts]
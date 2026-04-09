from pathlib import Path
from typing import List

def load_documents(raw_dir: str) -> List[str]:
    """Charge les documents bruts (simplifié : lit des .txt)."""
    texts = []
    for path in Path(raw_dir).glob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    return texts
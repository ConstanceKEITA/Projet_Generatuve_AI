from pathlib import Path
from typing import List


def load_documents(raw_dir: str) -> List[str]:
    """Charge les documents depuis data/raw/.
    Supporte : .txt, .pdf, .docx
    """
    texts = []
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")

    # Fichiers TXT
    for path in raw_path.glob("*.txt"):
        try:
            texts.append(path.read_text(encoding="utf-8"))
            print(f"  [TXT] {path.name}")
        except Exception as e:
            print(f"  [ERREUR TXT] {path.name} : {e}")

    # Fichiers PDF
    for path in raw_path.glob("*.pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                )
            texts.append(text)
            print(f"  [PDF] {path.name}")
        except ImportError:
            print("  [INFO] Installe pdfplumber : pip install pdfplumber")
        except Exception as e:
            print(f"  [ERREUR PDF] {path.name} : {e}")

    # Fichiers DOCX
    for path in raw_path.glob("*.docx"):
        try:
            import docx
            doc = docx.Document(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            texts.append(text)
            print(f"  [DOCX] {path.name}")
        except ImportError:
            print("  [INFO] Installe python-docx : pip install python-docx")
        except Exception as e:
            print(f"  [ERREUR DOCX] {path.name} : {e}")

    if not texts:
        print(f"  [ATTENTION] Aucun document trouvé dans {raw_dir}")

    return texts
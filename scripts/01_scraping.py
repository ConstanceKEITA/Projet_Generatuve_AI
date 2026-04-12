import json
import re
import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pdfplumber

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_FILE   = Path(__file__).parents[1] / "data" / "raw" / "ihl_raw.json"
REQUEST_DELAY = 1.5  # secondes entre chaque requête

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IHL-Scraper/1.0)"
}

# ── Sources officielles (français) ────────────────────────────────────────────
#
# fmt : "pdf" → téléchargement HTTP + extraction
#        "html" → scraping HTML

SOURCES = [
    {
        "name": "Statut de Rome de la Cour pénale internationale",
        "year": 1998,
        "type": "Statut constitutif",
        "url": "https://legal.un.org/icc/statute/french/rome_statute(f).pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["CPI", "Statut de Rome", "génocide",
                 "crimes contre l'humanité", "crimes de guerre",
                 "crime d'agression"],
    },
    {
        "name": "Conventions de Genève du 12 août 1949",
        "year": 1949,
        "type": "Convention internationale",
        "url": "https://elearning.icrc.org/healthcareindanger-legal-framework/fr/media/attachements/chapter-1/gva-conv.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["Conventions de Genève", "DIH", "crimes de guerre",
                 "protection des civils", "prisonniers de guerre"],
    },
    {
        "name": "Convention pour la prévention et la répression du crime de génocide",
        "year": 1948,
        "type": "Convention internationale",
        "url": "https://treaties.un.org/doc/Treaties/1951/01/19510112%2008-12%20PM/Ch_IV_1p.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["génocide", "ONU", "crime international"],
    },
    {
        "name": "Convention contre la torture et autres peines ou traitements cruels",
        "year": 1984,
        "type": "Convention internationale",
        "url": "https://treaties.un.org/doc/Publication/UNTS/Volume%201465/volume-1465-I-24841-french.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["torture", "traitement inhumain", "ONU", "droits de l'homme"],
    },
    {
        "name": "Statut du Tribunal pénal international pour l'ex-Yougoslavie",
        "year": 1993,
        "type": "Statut constitutif",
        "url": "https://www.icty.org/x/file/Legal%20Library/Statute/statute_sept09_fr.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["TPIY", "ex-Yougoslavie", "tribunal ad hoc", "crimes de guerre"],
    },
    {
        "name": "Statut du Tribunal pénal international pour le Rwanda",
        "year": 1994,
        "type": "Statut constitutif",
        "url": "https://unictr.irmct.org/sites/unictr.org/files/legal-library/100131_Statute_en_fr_0.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["TPIR", "Rwanda", "génocide", "tribunal ad hoc"],
    },
    {
        "name": "Principes de droit international reconnus par le Statut du Tribunal de Nuremberg",
        "year": 1950,
        "type": "Principes CDI",
        "url": "https://legal.un.org/ilc/texts/instruments/french/draft_articles/7_1_1950.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["Nuremberg", "CDI", "responsabilité individuelle", "crimes contre la paix"],
    },
    {
        "name": "Projet de Code des crimes contre la paix et la sécurité de l'humanité",
        "year": 1996,
        "type": "Projet CDI",
        "url": "https://legal.un.org/ilc/texts/instruments/french/draft_articles/7_4_1996.pdf",
        "fmt": "pdf",
        "lang": "fr",
        "tags": ["CDI", "code des crimes", "paix", "sécurité", "humanité"],
    },
]

# ── Fonctions de scraping ─────────────────────────────────────────────────────

def fetch_html(url: str) -> str | None:
    """Télécharge une page HTML et retourne le texte principal nettoyé."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        main = (
                soup.find("main")
                or soup.find("article")
                or soup.find(class_=lambda c: c and any(
            x in c.lower() for x in ["content", "main", "body", "article"]
        ))
                or soup.body
        )
        return main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    except requests.exceptions.HTTPError as e:
        log.warning(f"  ✗ HTTP {e.response.status_code} — {url}")
    except requests.exceptions.ConnectionError:
        log.warning(f"  ✗ Connexion impossible — {url}")
    except requests.exceptions.Timeout:
        log.warning(f"  ✗ Timeout — {url}")
    except Exception as e:
        log.warning(f"  ✗ Erreur inattendue ({url}): {e}")
    return None


def fetch_pdf(url: str) -> str | None:
    """Télécharge un PDF et extrait le texte de toutes ses pages."""
    tmp = Path("/tmp/ihl_tmp.pdf")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        tmp.write_bytes(resp.content)

        pages = []
        with pdfplumber.open(tmp) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages.append(txt)

        if not pages:
            log.warning(f"  ✗ Aucun texte extrait — {url}")
            return None

        return "\n".join(pages)

    except requests.exceptions.HTTPError as e:
        log.warning(f"  ✗ HTTP {e.response.status_code} — {url}")
    except requests.exceptions.Timeout:
        log.warning(f"  ✗ Timeout PDF — {url}")
    except Exception as e:
        log.warning(f"  ✗ Erreur PDF ({url}): {e}")
    finally:
        if tmp.exists():
            tmp.unlink()
    return None


def clean_text(text: str) -> str:
    """Nettoyage basique du texte brut."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def scrape_all() -> list[dict]:
    results = []

    for source in SOURCES:
        log.info(f"→ [{source['fmt'].upper()}] {source['name']}")
        time.sleep(REQUEST_DELAY)

        raw_text = fetch_pdf(source["url"]) if source["fmt"] == "pdf" else fetch_html(source["url"])

        if not raw_text:
            log.warning(f"  ⚠ Source ignorée (texte vide ou inaccessible)\n")
            continue

        clean = clean_text(raw_text)
        log.info(f"  ✓ {len(clean):,} caractères extraits\n")

        results.append({
            "name": source["name"],
            "year": source["year"],
            "type": source["type"],
            "url":  source["url"],
            "lang": source["lang"],
            "tags": source["tags"],
            "text": clean,
        })

    return results


def main():
    log.info("=== Démarrage du scraping ===\n")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    documents = scrape_all()

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    log.info(f"=== Terminé ===")
    log.info(f"✅ {len(documents)}/{len(SOURCES)} documents sauvegardés → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
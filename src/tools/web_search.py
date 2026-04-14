import logging
import os
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()
log = logging.getLogger(__name__)

LEGAL_SITES = [
    "ihl-databases.icrc.org",
    "icj-cij.org",
    "icc-cpi.int",
    "legal.un.org",
    "treaties.un.org",
    "icty.org",
    "unictr.irmct.org",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IHL-Assistant/1.0)"}


def fetch_page_text(url: str, max_chars: int = 2000) -> str:
    """Récupère le texte principal d'une page web."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text(separator="\n") if main else soup.get_text()
        return text[:max_chars]
    except Exception as e:
        log.warning(f"Impossible de fetcher {url} : {e}")
        return ""


def web_search(query: str, max_results: int = 3) -> str:
    """
    Recherche sur les bases juridiques internationales,
    extrait le contenu et synthétise avec Mistral.
    """
    try:
        # 1. Recherche DuckDuckGo
        site_filter = " OR ".join([f"site:{s}" for s in LEGAL_SITES])
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"({site_filter}) {query}",
                max_results=max_results,
                region="fr-fr"
            ))

        if not results:
            return f"Aucun résultat trouvé pour : '{query}'"

        # 2. Fetcher le contenu, fallback sur l'extrait DDG si 403
        context_parts = []
        sources = []
        for r in results:
            text = fetch_page_text(r["href"])
            if not text:
                text = r.get("body", "")  # ← fallback extrait DDG
            if text:
                context_parts.append(f"[{r['title']}]\n{text}")
                sources.append(f"{r['title']} → {r['href']}")

        if not context_parts:
            return "Impossible d'extraire le contenu des pages trouvées."

        context = "\n\n---\n\n".join(context_parts)

        # 3. Synthèse avec Mistral
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        prompt = f"""Tu es un assistant expert en Droit International Humanitaire.
En te basant uniquement sur les extraits ci-dessous, réponds en français de manière claire et précise à la question suivante.
Cite les sources pertinentes.

Question : {query}

Extraits :
{context}

Réponse :"""

        response = llm.invoke([HumanMessage(content=prompt)])

        answer = response.content

        return answer, sources

    except Exception as e:
        log.warning(f"Erreur web_search : {e}")
        return f"Erreur lors de la recherche : {e}"


if __name__ == "__main__":
    print(web_search("mandat d'arrêt CPI Poutine Ukraine"))
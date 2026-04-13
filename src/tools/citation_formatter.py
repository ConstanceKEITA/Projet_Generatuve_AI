import logging
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()
log = logging.getLogger(__name__)


def format_citation(reference: str) -> str:
    """
    Formate une référence juridique au standard académique international.
    """
    if not reference or len(reference.strip()) < 5:
        return "Référence trop courte ou vide."

    try:
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        prompt = f"""Tu es un expert en citation juridique internationale.
Formate la référence suivante selon les standards académiques internationaux (RTNU, CICR, CIJ, CPI).
Inclus : titre complet, date, lieu d'adoption, numéro RTNU si applicable, article si mentionné.
Réponds uniquement avec la citation formatée, sans explication.

Référence : {reference}

Citation formatée :"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    except Exception as e:
        log.warning(f"Erreur citation_formatter : {e}")
        return f"Erreur lors du formatage : {e}"


if __name__ == "__main__":
    exemples = [
        "Convention de Genève IV, art. 3",
        "Statut de Rome, article 8",
        "Convention génocide 1948",
        "Protocole additionnel I, art. 51",
    ]
    for ex in exemples:
        print(f"Input : {ex}")
        print(f"Output : {format_citation(ex)}\n")
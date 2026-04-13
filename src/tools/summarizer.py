import logging
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()
log = logging.getLogger(__name__)


def summarize(text: str, max_points: int = 5) -> str:
    """
    Résume un texte juridique en points clés via Mistral.
    """
    if not text or len(text.strip()) < 50:
        return "Texte trop court ou vide pour être résumé."

    try:
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        prompt = f"""Tu es un assistant expert en Droit International Humanitaire.
Résume le texte juridique suivant en {max_points} points clés maximum.
Sois précis, concis et conserve les références aux articles et conventions.
Réponds en français.

Texte :
{text}

Résumé en points clés :"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    except Exception as e:
        log.warning(f"Erreur summarizer : {e}")
        return f"Erreur lors du résumé : {e}"


if __name__ == "__main__":
    test = """
    Article 3 commun aux Conventions de Genève.
    En cas de conflit armé ne présentant pas un caractère international
    et surgissant sur le territoire de l'une des Hautes Parties contractantes,
    chacune des Parties au conflit sera tenue d'appliquer au moins les dispositions suivantes:
    1) Les personnes qui ne participent pas directement aux hostilités,
    y compris les membres de forces armées qui ont déposé les armes
    et les personnes qui ont été mises hors de combat par maladie, blessure,
    détention, ou pour toute autre cause, seront, en toutes circonstances,
    traitées avec humanité, sans aucune distinction de caractère défavorable
    basée sur la race, la couleur, la religion ou la croyance, le sexe,
    la naissance ou la fortune, ou tout autre critère analogue.
    """
    print(summarize(test))
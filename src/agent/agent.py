import re
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from tools.calculator import calculate
from tools.weather import get_weather
from tools.web_search import web_search
from tools.summarizer import summarize
from tools.citation_formatter import format_citation


def extract_city(query: str, history: str = "") -> str:
    """Extrait le nom de la ville depuis la question, avec contexte conversationnel."""
    # Essayer d'abord avec le regex sur la query seule
    q = query.lower()
    patterns = [
        r"météo (?:à|a|au|en|de|pour)\s+([a-zA-ZÀ-ÿ\s\-]+?)(?:\s*\?|$)",
        r"meteo (?:à|a|au|en|de|pour)\s+([a-zA-ZÀ-ÿ\s\-]+?)(?:\s*\?|$)",
        r"temps (?:à|a|au|en|de|pour)\s+([a-zA-ZÀ-ÿ\s\-]+?)(?:\s*\?|$)",
        r"température (?:à|a|au|en|de|pour)\s+([a-zA-ZÀ-ÿ\s\-]+?)(?:\s*\?|$)",
        r"weather (?:in|at|for)\s+([a-zA-ZÀ-ÿ\s\-]+?)(?:\s*\?|$)",
        r"(?:à|a|au|en)\s+([a-zA-ZÀ-ÿ\s\-]+?)(?:\s*\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match.group(1).strip().title()

    # Fallback : utiliser Mistral avec le contexte conversationnel
    if history:
        llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
        prompt = f"""En te basant sur l'historique de conversation et la question, extrait le nom de la ville pour laquelle on demande la météo.
Réponds UNIQUEMENT avec le nom de la ville en anglais, sans ponctuation ni explication.
Si aucune ville n'est trouvée, réponds "Paris".
 
Historique :
{history}
 
Question : {query}
Ville :"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().title()

    return "Paris"


class Agent:
    def __init__(self, rag_callable):
        self.rag_callable = rag_callable
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0
        )

    def _route(self, query: str, history: str = "") -> str:
        """Demande à Mistral quel outil utiliser."""
        history_section = f"\nHistorique récent :\n{history}\n" if history else ""

        routing_prompt = f"""Tu es un router d'outils pour un assistant juridique spécialisé en Droit Pénal International.
Analyse la question et réponds UNIQUEMENT par un de ces mots, sans ponctuation ni explication :
 
- RAG : question théorique sur des concepts juridiques, articles, conventions, définitions, principes, obligations du Droit Pénal International (ex: "qu'est-ce que le principe de distinction ?", "que dit l'article 3 commun ?", "quelles sont les obligations des parties à un conflit ?")
- WEB : question sur des faits réels, des personnes nommées, des événements datés, des actualités, des mandats d'arrêt, des décisions récentes de tribunaux, des conflits en cours (ex: "mandat CPI contre Poutine", "dernières décisions de la CIJ sur Gaza", "Netanyahou a-t-il été arrêté ?")
- METEO : question sur la météo actuelle d'une ville ou d'un pays (ex: "quel temps fait-il à Genève ?")
- CALCUL : opération mathématique pure avec des chiffres (ex: "combien font 12 * 8 ?", "1945 + 79 ?")
- RESUME : l'utilisateur demande explicitement un résumé, une synthèse ou les points clés d'un texte, d'un article juridique ou d'une convention (ex: "résume l'article 51", "quels sont les points clés du Protocole additionnel I ?", "synthétise la Convention de Genève IV")
- CITATION : l'utilisateur veut formater ou mettre en forme une référence juridique selon les standards académiques (ex: "formate la Convention de Genève III art. 17", "cite le Statut de Rome article 8")
- CHAT : salutation, remerciement, question générale sans lien avec le Droit Pénal International, conversation informelle (ex: "Bonjour", "Merci", "Comment vas-tu ?", "Tu peux m'aider ?", "C'est quoi ton nom ?")
 
En cas de doute entre RAG et WEB, privilégie WEB si la question porte sur un événement daté, une personne nommée, ou un fait vérifiable récent.
En cas de doute entre RAG et RESUME, privilégie RESUME si l'utilisateur utilise explicitement les mots "résume", "synthétise", "points clés", "en bref" ou "simplifie".
En cas de doute entre RESUME et WEB, privilégie RESUME si l'utilisateur cite un article ou une convention précise.
En cas de doute entre RAG et CITATION, privilégie CITATION si l'utilisateur utilise les mots "formate", "cite", "référence" ou "bibliographie".
En cas de doute entre WEB et CITATION, privilégie CITATION si la question porte sur la mise en forme d'une source juridique plutôt que sur son contenu.
Si aucun outil ne correspond clairement, utilise RAG par défaut pour maximiser l'exhaustivité de la réponse.
Si la question est une salutation ou une question générale sans lien avec le droit, utilise CHAT.
{history_section}
Question : {query}
Outil :"""

        response = self.llm.invoke([HumanMessage(content=routing_prompt)])
        return response.content.strip().upper()

    def _is_rag_sufficient(self, rag_response: str, query: str) -> bool:
        check_prompt = f"""Tu es un évaluateur de réponses juridiques.
La question posée est : {query}
La réponse RAG obtenue est : {rag_response}
 
Cette réponse est-elle suffisante et précise pour répondre à la question ?
Réponds UNIQUEMENT par OUI ou NON.
"""
        response = self.llm.invoke([HumanMessage(content=check_prompt)])
        return "OUI" in response.content.upper()

    def _extract_cities(self, query: str) -> list[str]:
        prompt = f"""Extrait tous les noms de villes mentionnés dans cette question.
    Réponds UNIQUEMENT avec les noms de villes séparés par des virgules, sans ponctuation ni explication.
    Si la question mentionne un pays, donne sa grande ville principale.
    Si aucune ville n'est mentionnée, réponds "Paris".
    
    Exemples :
    - "météo à Paris et Lyon ?" → Paris, Lyon
    - "quel temps à Londres et Berlin ?" → London, Berlin
    - "météo au Maroc ?" → Casablanca
    
    Question : {query}
    Villes :"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        cities = [c.strip() for c in response.content.split(",")]
        return cities

    def decide_and_answer(self, query: str, history: str = "") -> tuple[str, list, str]:
        tool = self._route(query, history)

        if tool == "WEB":
            enriched_query = f"{history}\n\nQuestion : {query}" if history else query
            answer, sources = web_search(enriched_query)
            return answer, sources, "🌐 Web Search"
        elif tool == "METEO":
            city = extract_city(query, history)
            return get_weather(city), [], "🌤️ Météo"
        elif tool == "CALCUL":
            return calculate(query), [], "🔢 Calcul"
        elif tool == "RESUME":
            rag_content, sources = self.rag_callable(query, history=history)
            return summarize(rag_content), sources, "📝 Résumé"
        elif tool == "CITATION":
            return format_citation(query), [], "📌 Citation"
        elif tool == "CHAT":
            chat_prompt = f"{history}\n\nUtilisateur : {query}" if history else query
            response = self.llm.invoke([HumanMessage(content=chat_prompt)])
            return response.content, [], "💬 Chat"
        else:
            rag_response, sources = self.rag_callable(query, history=history)
            if self._is_rag_sufficient(rag_response, query):
                return rag_response, sources, "📄 RAG"
            else:
                web_response, web_sources = web_search(query)
                return web_response, sources + web_sources, "⚖️ RAG + Web"
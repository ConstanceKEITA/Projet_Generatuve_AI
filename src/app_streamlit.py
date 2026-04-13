import streamlit as st
from router import Router

def detect_tool(query: str) -> str:
    """Détecte quel outil a été utilisé pour afficher le badge."""
    q = query.lower()
    if "météo" in q or "meteo" in q:
        return "Météo"
    if "résume" in q or "résumer" in q:
        return "Résumé"
    if "citation" in q or "formate" in q:
        return "Citation"
    if "recherche" in q or "web" in q:
        return "Web Search"
    return "RAG"

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant DIH",
    page_icon="⚖️",
    layout="wide"
)

# ── Style CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1a3a5c;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .main { background-color: #ffffff; }
    .user-bubble {
        background-color: #1a3a5c;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .assistant-bubble {
        background-color: #eaf4fb;
        color: #2c3e50;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 80%;
        border: 1px solid #2980b9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .tool-badge {
        background-color: #2980b9;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 12px;
        margin-bottom: 5px;
        display: inline-block;
    }
     .stTextInput input {
        border-radius: 20px;
        border: 2px solid #f39c12;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button {
            background-color: #f39c12;
            color: white;
            border-radius: 20px;
            font-weight: bold;
    }
""", unsafe_allow_html=True)

# ── Style CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    ... # ton CSS existant
</style>
""", unsafe_allow_html=True)

# ← ajoute ici
st.markdown("""
<style>
    [data-testid="stSidebar"] img {
        filter: brightness(0) invert(1);
    }
</style>
""", unsafe_allow_html=True)

# ── Initialisation ────────────────────────────────────────────────────────────
if "router" not in st.session_state:
    with st.spinner("Chargement de l'assistant..."):
        st.session_state.router = Router()
        st.session_state.history = []

router = st.session_state.router

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("src/assets/logo.png", width=80)
    st.title("⚖️ Assistant DIH")
    st.markdown("---")

    st.markdown("### 🛠️ Outils disponibles")
    st.markdown("""
    - 📄 **RAG** — Documents DIH
    - 🌐 **Web Search** — Bases juridiques
    - 🧮 **Calculatrice**
    - 🌤️ **Météo**
    - 📝 **Résumé juridique**
    - 📌 **Citation formatter**
    """)

    st.markdown("---")
    st.markdown("### 💡 Exemples de questions")
    st.markdown("""
    - *Qu'est-ce que le génocide ?*
    - *résume Article 3 des Conventions de Genève...*
    - *formate Convention de Genève IV, art. 3*
    - *Poutine a reçu un mandat d'arrêt ?*
    - *quelle est la météo à Paris ?*
    - *2 + 2*
    """)

    st.markdown("---")
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.session_state.router = Router()
        st.rerun()

# ── Interface principale ──────────────────────────────────────────────────────
st.title("⚖️ Assistant Intelligent — Droit International Humanitaire")
st.caption("Combine RAG, agents juridiques et mémoire conversationnelle")

# Affichage de l'historique
chat_container = st.container()
with chat_container:
    for speaker, msg, tool in st.session_state.history:
        if speaker == "Vous":
            st.markdown(f'<div class="user-bubble">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            badge = f'<span class="tool-badge">🔧 {tool}</span><br>' if tool else ""
            st.markdown(
                f'<div class="assistant-bubble">{badge}🤖 {msg}</div>',
                unsafe_allow_html=True
            )

# Champ de saisie
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Pose ta question :",
        placeholder="Ex : Quelle est la définition du génocide ?",
        label_visibility="collapsed"
    )
with col2:
    submit = st.button("Envoyer →", use_container_width=True)

if (query and submit) or query:
    with st.spinner("Réflexion en cours..."):
        answer = router.route(query)
        tool_used = detect_tool(query)

    st.session_state.history.append(("Vous", query, None))
    st.session_state.history.append(("Assistant", answer, tool_used))
    st.rerun()

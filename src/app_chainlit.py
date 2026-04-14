import chainlit as cl
from router import Router

def detect_tool(query: str) -> str:
    q = query.lower()
    if "météo" in q or "meteo" in q:
        return "🌤️ Météo"
    if "résume" in q or "résumer" in q:
        return "📝 Résumé"
    if "citation" in q or "formate" in q:
        return "📌 Citation"
    if "recherche" in q or "web" in q:
        return "🌐 Web Search"
    return "📄 RAG"

@cl.on_chat_start
async def on_start():
    router = Router()
    cl.user_session.set("router", router)
    await cl.Message(
        content="👋 Bonjour ! Je suis l'**Assistant DIH**. Posez-moi une question sur le Droit International Humanitaire.",
        author="Assistant DIH️"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    router = cl.user_session.get("router")
    query = message.content
    tool_used = detect_tool(query)

    async with cl.Step(name=tool_used) as step:
        step.input = query
        answer, sources = router.route(query)
        step.output = answer

    elements = []
    if sources:
        sources_text = "\n".join(f"• {s}" for s in sources)
        elements.append(
            cl.Text(
                name="📚 Sources",
                content=sources_text,
                display="side"
            )
        )

    await cl.Message(
        content=answer,
        elements=elements,
        author="Assistant DIH"
    ).send()
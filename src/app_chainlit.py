import base64
import chainlit as cl
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router import Router


async def describe_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    llm = ChatMistralAI(model="pixtral-12b-2409", temperature=0)
    response = llm.invoke([HumanMessage(content=[
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"},
        {"type": "text", "text": "Décris le contenu de cette image en français de manière précise."}
    ])])
    return response.content


@cl.on_chat_start
async def on_start():
    router = Router()
    cl.user_session.set("router", router)
    await cl.Message(
        content= "⚖️ Bonjour, je suis **Aequitas**. Posez-moi vos questions en Droit Pénal International — je consulte vos documents, le web juridique, et bien plus.",
        author="Aequitas"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    router = cl.user_session.get("router")
    query = message.content

    # Gestion des images jointes
    if message.elements:
        for element in message.elements:
            if hasattr(element, "mime") and "image" in element.mime:
                image_description = await describe_image(element.path)
                query = f"{query}\n\nContenu de l'image : {image_description}" if query else image_description

    answer, sources, tool_used = router.route(query)

    async with cl.Step(name=tool_used) as step:
        step.input = query
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
        author="Aequitas"
    ).send()
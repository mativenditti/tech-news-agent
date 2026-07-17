"""Briefing proactivo: el 'top 3 titulares' del lunes a la mañana.

Invoca el mismo grafo con el prompt de briefing. Al correr por el grafo, hereda
guardrails, tools (web_search), tracing de LangSmith y memoria por thread.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import graph
from app.prompts import BRIEFING_PROMPT


def run_briefing(thread_id: str | None = None, user_role: str | None = None) -> dict:
    """Genera el briefing y devuelve {thread_id, briefing}.

    Usa un thread nuevo por defecto para que el usuario pueda seguir la
    conversación (elegir un titular para profundizar) en ese mismo thread.
    """
    thread_id = thread_id or f"briefing-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=BRIEFING_PROMPT)],
            "user_role": user_role,
            "tool_call_counts": {},
            "blocked": False,
        },
        config=config,
    )

    # Gemini 3 devuelve content como lista de bloques, no string. AIMessage.text
    # extrae el texto en ambos casos; chequear isinstance(content, str) daría "".
    text = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.text:
            text = msg.text
            break

    return {"thread_id": thread_id, "briefing": text}

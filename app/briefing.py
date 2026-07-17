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


def stream_briefing(thread_id: str | None = None, user_role: str | None = None):
    """Streamea el briefing proactivo como eventos SSE (dicts {"event","data"}).

    Reusa el loop común de streaming (stream_chat/_stream_graph_events); solo cambia
    el prompt de entrada y el prefijo del thread_id. Emite `token`* y luego `done`
    (o `approval` si hubiera un interrupt; el briefing no dispara el email, pero se
    maneja igual por consistencia). Los errores se traducen a un evento `error`.
    """
    from app.chat import _stream_graph_events

    thread_id = thread_id or f"briefing-{uuid.uuid4().hex[:8]}"
    inputs = {
        "messages": [HumanMessage(content=BRIEFING_PROMPT)],
        "user_role": user_role,
        "tool_call_counts": {},
        "blocked": False,
    }
    yield from _stream_graph_events(inputs, thread_id)
